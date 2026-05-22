import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Plus, Upload, Loader2, CheckCircle, Building2, Inbox, Eye, AlertCircle, TrendingDown, Wallet, Sparkles, FileText, X } from "lucide-react";
import { toast } from "sonner";
import { useServerFn } from "@tanstack/react-start";
import { ocrFacture } from "@/server/factures.functions";

export const Route = createFileRoute("/_app/dossiers/$dossierId/fournisseurs")({ component: FournisseursPage });

interface FactureF {
  id: string; fournisseur_nom: string | null; fournisseur_id: string | null;
  numero: string | null; date_facture: string | null; date_echeance: string | null;
  montant_ht: number; montant_tva: number; montant_ttc: number;
  montant_paye: number; montant_restant: number;
  statut: string; statut_paiement: string; mode_reglement: string | null;
}

const fmt = (n: number) => Number(n).toLocaleString("fr-MA", { minimumFractionDigits: 2 }) + " MAD";

const MODES = ["virement", "cheque", "especes", "carte", "prelevement"];

function FournisseursPage() {
  const { dossierId } = Route.useParams();
  const ocrFn = useServerFn(ocrFacture);

  const [tab, setTab] = useState<"factures"|"saisie"|"tiers">("factures");
  const [factures, setFactures] = useState<FactureF[]>([]);
  const [fournisseurs, setFournisseurs] = useState<any[]>([]);
  const [dossier, setDossier] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState<string|null>(null);

  // OCR
  const [ocrLoading, setOcrLoading] = useState(false);
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState<string|null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Formulaire facture
  const [fournisseurId, setFournisseurId] = useState("");
  const [fournisseurNom, setFournisseurNom] = useState("");
  const [fournisseurIce, setFournisseurIce] = useState("");
  const [numero, setNumero] = useState("");
  const [dateFacture, setDateFacture] = useState(new Date().toISOString().slice(0,10));
  const [dateEcheance, setDateEcheance] = useState("");
  const [montantHt, setMontantHt] = useState(0);
  const [montantTva, setMontantTva] = useState(0);
  const [montantTtc, setMontantTtc] = useState(0);
  const [modeReglement, setModeReglement] = useState("virement");
  const [lignes, setLignes] = useState<any[]>([{ designation: "", quantite: 1, prix_unitaire: 0, taux_tva: 20 }]);

  // Modal fournisseur
  const [openFourn, setOpenFourn] = useState(false);
  const [formFourn, setFormFourn] = useState({ nom: "", ice: "", email: "", telephone: "" });

  const load = async () => {
    setLoading(true);
    const [{ data: ff }, { data: fs }, { data: dos }] = await Promise.all([
      (supabase.from("factures_fournisseurs") as any).select("*").eq("dossier_id", dossierId).order("created_at", { ascending: false }),
      supabase.from("fournisseurs").select("*").eq("dossier_id", dossierId).order("nom"),
      (supabase.from("dossiers") as any).select("nom_societe,ice").eq("id", dossierId).single(),
    ]);
    setFactures((ff ?? []) as FactureF[]);
    setFournisseurs(fs ?? []);
    setDossier(dos);
    setLoading(false);
  };

  useEffect(() => { load(); }, [dossierId]);

  // Recalculer TTC quand lignes changent
  useEffect(() => {
    const ht = lignes.reduce((s, l) => s + l.quantite * l.prix_unitaire, 0);
    const tva = lignes.reduce((s, l) => s + l.quantite * l.prix_unitaire * l.taux_tva / 100, 0);
    setMontantHt(Math.round(ht * 100) / 100);
    setMontantTva(Math.round(tva * 100) / 100);
    setMontantTtc(Math.round((ht + tva) * 100) / 100);
  }, [lignes]);

  // ── OCR PDF ────────────────────────────────────────────────────────────────
  const handleOcr = async (file: File) => {
    setOcrLoading(true);
    setPdfPreviewUrl(URL.createObjectURL(file));
    try {
      // Extraction texte PDF
      const pdfjsLib = await import("pdfjs-dist");
      pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;
      const ab = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: ab }).promise;
      let text = "";
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const items = content.items as any[];
        let lastY = -1, line = "";
        for (const item of items) {
          const y = Math.round(item.transform[5]);
          if (lastY !== -1 && Math.abs(y - lastY) > 3) { text += line.trimEnd() + "\n"; line = ""; }
          line += item.str + " "; lastY = y;
        }
        if (line.trim()) text += line.trimEnd() + "\n";
      }

      // Appel OCR IA
      const { result: r } = await ocrFn({ data: { extracted_text: text, dossier_id: dossierId } });

      // Remplir le formulaire
      if (r.client_nom_extrait) setFournisseurNom(r.emetteur_nom || r.client_nom_extrait);
      if (r.emetteur_nom) setFournisseurNom(r.emetteur_nom);
      if (r.emetteur_ice) setFournisseurIce(r.emetteur_ice);
      if (r.numero_facture) setNumero(r.numero_facture);
      if (r.date_facture) setDateFacture(r.date_facture);
      if (r.date_echeance) setDateEcheance(r.date_echeance);
      if (r.montant_ht) setMontantHt(r.montant_ht);
      if (r.montant_tva) setMontantTva(r.montant_tva);
      if (r.montant_ttc) setMontantTtc(r.montant_ttc);
      if (r.mode_reglement) setModeReglement(r.mode_reglement);
      if (r.lignes?.length) setLignes(r.lignes.map((l: any) => ({
        designation: l.designation,
        quantite: l.quantite,
        prix_unitaire: l.prix_unitaire,
        taux_tva: l.taux_tva,
      })));

      // Chercher fournisseur existant
      if (r.emetteur_ice) {
        const found = fournisseurs.find(f => f.ice === r.emetteur_ice);
        if (found) setFournisseurId(found.id);
      } else if (r.emetteur_nom) {
        const found = fournisseurs.find(f => f.nom?.toLowerCase().includes(r.emetteur_nom?.toLowerCase()?.slice(0, 5)));
        if (found) setFournisseurId(found.id);
      }

      setTab("saisie");
      toast.success("OCR terminé — vérifiez et complétez les données");
    } catch (e: any) {
      toast.error("Erreur OCR: " + e.message);
    } finally {
      setOcrLoading(false);
    }
  };

  // ── Enregistrer facture fournisseur ────────────────────────────────────────
  const handleSave = async () => {
    if (!fournisseurNom && !fournisseurId) return toast.error("Sélectionnez ou saisissez un fournisseur");
    if (!montantTtc) return toast.error("Montant TTC requis");
    setProcessing("save");
    try {
      // Créer fournisseur si nouveau
      let fId = fournisseurId;
      if (!fId && fournisseurNom) {
        const { data: nouveau } = await supabase.from("fournisseurs").insert({
          dossier_id: dossierId, nom: fournisseurNom, ice: fournisseurIce || null,
        }).select("id").single();
        if (nouveau) fId = nouveau.id;
      }

      const nomFourn = fId ? fournisseurs.find(f => f.id === fId)?.nom || fournisseurNom : fournisseurNom;

      const { error } = await (supabase.from("factures_fournisseurs") as any).insert({
        dossier_id: dossierId,
        fournisseur_id: fId || null,
        fournisseur_nom: nomFourn,
        numero: numero || null,
        date_facture: dateFacture,
        date_echeance: dateEcheance || null,
        montant_ht: montantHt,
        montant_tva: montantTva,
        montant_ttc: montantTtc,
        montant_paye: 0,
        montant_restant: montantTtc,
        statut: "recue",
        statut_paiement: "non_payee",
        mode_reglement: modeReglement,
        lignes: lignes,
      });
      if (error) throw error;

      // Écritures comptables ACH
      const ref = numero || nomFourn;
      await supabase.from("ecritures_comptables").insert([
        { dossier_id: dossierId, journal_code: "ACH", compte_numero: "6141", date_ecriture: dateFacture, libelle: `Achat ${nomFourn} ${ref}`, debit: montantHt, credit: 0, valide: true },
        { dossier_id: dossierId, journal_code: "ACH", compte_numero: "34552", date_ecriture: dateFacture, libelle: `TVA ${nomFourn}`, debit: montantTva, credit: 0, valide: true },
        { dossier_id: dossierId, journal_code: "ACH", compte_numero: "4411", date_ecriture: dateFacture, libelle: `Dette ${nomFourn}`, debit: 0, credit: montantTtc, valide: true },
      ]);

      toast.success("Facture fournisseur enregistrée ✅");
      resetForm();
      load();
      setTab("factures");
    } catch (e: any) { toast.error(e.message); }
    finally { setProcessing(null); }
  };

  const resetForm = () => {
    setFournisseurId(""); setFournisseurNom(""); setFournisseurIce("");
    setNumero(""); setDateFacture(new Date().toISOString().slice(0,10));
    setDateEcheance(""); setMontantHt(0); setMontantTva(0); setMontantTtc(0);
    setModeReglement("virement"); setPdfPreviewUrl(null);
    setLignes([{ designation: "", quantite: 1, prix_unitaire: 0, taux_tva: 20 }]);
  };

  // KPIs
  const dettes = factures.filter(f => f.statut_paiement !== "payee").reduce((s, f) => s + Number(f.montant_restant ?? f.montant_ttc), 0);
  const depenses = factures.filter(f => f.statut_paiement !== "payee").reduce((s, f) => s + Number(f.montant_ht), 0);
  const enAttente = factures.filter(f => f.statut_paiement !== "payee").length;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Factures fournisseurs</h1>
          <p className="text-muted-foreground mt-1">Import OCR · Saisie manuelle · Comptabilisation automatique</p>
        </div>
        <div className="flex gap-2">
          <input ref={fileRef} type="file" accept=".pdf,.jpg,.jpeg,.png" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleOcr(f); }} />
          <Button variant="outline" onClick={() => fileRef.current?.click()} disabled={ocrLoading}>
            {ocrLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Sparkles className="h-4 w-4 mr-2" />}
            Scanner facture (OCR)
          </Button>
          <Button onClick={() => { resetForm(); setTab("saisie"); }}>
            <Plus className="h-4 w-4 mr-2" />Saisie manuelle
          </Button>
          <Button variant="ghost" onClick={() => setOpenFourn(true)}>
            <Building2 className="h-4 w-4 mr-1" />Fournisseur
          </Button>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {[
          { label: "Dettes fournisseurs", value: fmt(dettes), icon: Wallet, color: "text-red-600" },
          { label: "Achats HT en attente", value: fmt(depenses), icon: TrendingDown, color: "text-orange-600" },
          { label: "Factures non payées", value: String(enAttente), icon: AlertCircle, color: "text-yellow-600" },
        ].map(k => (
          <Card key={k.label}><CardContent className="pt-4 pb-4 flex items-center justify-between">
            <div>
              <p className="text-xs text-muted-foreground">{k.label}</p>
              <p className={`text-xl font-bold mt-1 ${k.color}`}>{k.value}</p>
            </div>
            <k.icon className={`h-8 w-8 ${k.color} opacity-30`} />
          </CardContent></Card>
        ))}
      </div>

      <Tabs value={tab} onValueChange={v => setTab(v as any)}>
        <TabsList>
          <TabsTrigger value="factures">Factures ({factures.length})</TabsTrigger>
          <TabsTrigger value="saisie">
            {ocrLoading ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <FileText className="h-3 w-3 mr-1" />}
            Saisie / OCR
          </TabsTrigger>
          <TabsTrigger value="tiers">Fournisseurs ({fournisseurs.length})</TabsTrigger>
        </TabsList>

        {/* ── Liste factures ── */}
        <TabsContent value="factures" className="mt-4">
          <Card><CardContent className="p-0">
            <Table>
              <TableHeader><TableRow>
                <TableHead>Fournisseur</TableHead><TableHead>N°</TableHead><TableHead>Date</TableHead>
                <TableHead className="text-right">HT</TableHead><TableHead className="text-right">TTC</TableHead>
                <TableHead className="text-right">Restant</TableHead><TableHead>Statut</TableHead>
              </TableRow></TableHeader>
              <TableBody>
                {loading
                  ? <TableRow><TableCell colSpan={7} className="text-center py-8"><Loader2 className="h-5 w-5 animate-spin mx-auto" /></TableCell></TableRow>
                  : factures.length === 0
                  ? <TableRow><TableCell colSpan={7} className="text-center py-12 text-muted-foreground">
                      <Inbox className="h-8 w-8 mx-auto mb-2 opacity-30" />
                      Aucune facture — scannez un PDF ou faites une saisie manuelle
                    </TableCell></TableRow>
                  : factures.map(f => (
                    <TableRow key={f.id}>
                      <TableCell className="font-medium">{f.fournisseur_nom}</TableCell>
                      <TableCell className="font-mono text-xs">{f.numero ?? "—"}</TableCell>
                      <TableCell className="text-sm">{f.date_facture ? new Date(f.date_facture).toLocaleDateString("fr-MA") : "—"}</TableCell>
                      <TableCell className="font-mono text-sm text-right">{fmt(Number(f.montant_ht))}</TableCell>
                      <TableCell className="font-mono text-sm text-right font-medium">{fmt(Number(f.montant_ttc))}</TableCell>
                      <TableCell className="font-mono text-sm text-right text-red-600">
                        {f.statut_paiement !== "payee" ? fmt(Number(f.montant_restant ?? f.montant_ttc)) : "—"}
                      </TableCell>
                      <TableCell>
                        <Badge variant={f.statut_paiement === "payee" ? "default" : f.statut_paiement === "partielle" ? "secondary" : "outline"}>
                          {f.statut_paiement === "payee" ? "✅ Payée" : f.statut_paiement === "partielle" ? "🔵 Partielle" : "⏳ En attente"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </CardContent></Card>
        </TabsContent>

        {/* ── Saisie / OCR ── */}
        <TabsContent value="saisie" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Aperçu PDF */}
            {pdfPreviewUrl && (
              <div className="relative">
                <Button size="sm" variant="ghost" className="absolute top-2 right-2 z-10"
                  onClick={() => setPdfPreviewUrl(null)}><X className="h-4 w-4" /></Button>
                <iframe src={pdfPreviewUrl} className="w-full h-[600px] rounded border" />
              </div>
            )}

            {/* Formulaire */}
            <div className="space-y-4">
              {/* Upload PDF */}
              {!pdfPreviewUrl && (
                <Card className="border-dashed cursor-pointer hover:border-primary transition-colors"
                  onClick={() => fileRef.current?.click()}>
                  <CardContent className="py-8 text-center">
                    {ocrLoading
                      ? <><Loader2 className="h-8 w-8 animate-spin mx-auto mb-2 text-primary" /><p>OCR en cours…</p></>
                      : <><Sparkles className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                        <p className="font-medium">Scanner une facture PDF</p>
                        <p className="text-sm text-muted-foreground mt-1">Cliquez ou glissez le PDF ici</p></>
                    }
                  </CardContent>
                </Card>
              )}

              {/* Fournisseur */}
              <div className="space-y-2">
                <Label>Fournisseur *</Label>
                <Select value={fournisseurId} onValueChange={v => {
                  setFournisseurId(v);
                  const f = fournisseurs.find(f => f.id === v);
                  if (f) { setFournisseurNom(f.nom); setFournisseurIce(f.ice || ""); }
                }}>
                  <SelectTrigger><SelectValue placeholder="Sélectionner un fournisseur…" /></SelectTrigger>
                  <SelectContent>
                    {fournisseurs.map(f => <SelectItem key={f.id} value={f.id}>{f.nom}</SelectItem>)}
                  </SelectContent>
                </Select>
                <Input placeholder="Ou saisir le nom du fournisseur" value={fournisseurNom}
                  onChange={e => { setFournisseurNom(e.target.value); setFournisseurId(""); }} />
                {!fournisseurId && fournisseurNom && (
                  <Input placeholder="ICE fournisseur" value={fournisseurIce}
                    onChange={e => setFournisseurIce(e.target.value)} />
                )}
              </div>

              {/* Infos facture */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2"><Label>N° facture</Label>
                  <Input value={numero} onChange={e => setNumero(e.target.value)} placeholder="FAC-2026-001" />
                </div>
                <div className="space-y-2"><Label>Mode règlement</Label>
                  <Select value={modeReglement} onValueChange={setModeReglement}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>{MODES.map(m => <SelectItem key={m} value={m}>{m}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
                <div className="space-y-2"><Label>Date facture *</Label>
                  <Input type="date" value={dateFacture} onChange={e => setDateFacture(e.target.value)} />
                </div>
                <div className="space-y-2"><Label>Date échéance</Label>
                  <Input type="date" value={dateEcheance} onChange={e => setDateEcheance(e.target.value)} />
                </div>
              </div>

              {/* Lignes */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Lignes de la facture</Label>
                  <Button size="sm" variant="outline" onClick={() => setLignes([...lignes, { designation: "", quantite: 1, prix_unitaire: 0, taux_tva: 20 }])}>
                    <Plus className="h-3 w-3 mr-1" />Ligne
                  </Button>
                </div>
                <div className="space-y-2">
                  {lignes.map((l, i) => (
                    <div key={i} className="grid grid-cols-12 gap-1 items-center">
                      <Input className="col-span-5 text-xs" placeholder="Désignation" value={l.designation}
                        onChange={e => setLignes(ls => ls.map((x, j) => j === i ? { ...x, designation: e.target.value } : x))} />
                      <Input className="col-span-2 text-xs" type="number" placeholder="Qté" value={l.quantite}
                        onChange={e => setLignes(ls => ls.map((x, j) => j === i ? { ...x, quantite: parseFloat(e.target.value) || 1 } : x))} />
                      <Input className="col-span-2 text-xs" type="number" placeholder="PU HT" value={l.prix_unitaire}
                        onChange={e => setLignes(ls => ls.map((x, j) => j === i ? { ...x, prix_unitaire: parseFloat(e.target.value) || 0 } : x))} />
                      <Select value={String(l.taux_tva)} onValueChange={v => setLignes(ls => ls.map((x, j) => j === i ? { ...x, taux_tva: parseInt(v) } : x))}>
                        <SelectTrigger className="col-span-2 text-xs h-9"><SelectValue /></SelectTrigger>
                        <SelectContent><SelectItem value="0">0%</SelectItem><SelectItem value="7">7%</SelectItem><SelectItem value="10">10%</SelectItem><SelectItem value="14">14%</SelectItem><SelectItem value="20">20%</SelectItem></SelectContent>
                      </Select>
                      <Button size="sm" variant="ghost" className="col-span-1 h-9 px-1"
                        onClick={() => setLignes(ls => ls.filter((_, j) => j !== i))}><X className="h-3 w-3" /></Button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Totaux */}
              <Card className="bg-muted/40">
                <CardContent className="pt-3 pb-3">
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div><p className="text-xs text-muted-foreground">HT</p><p className="font-bold">{fmt(montantHt)}</p></div>
                    <div><p className="text-xs text-muted-foreground">TVA</p><p className="font-bold">{fmt(montantTva)}</p></div>
                    <div><p className="text-xs text-muted-foreground">TTC</p><p className="font-bold text-primary">{fmt(montantTtc)}</p></div>
                  </div>
                </CardContent>
              </Card>

              <div className="flex gap-2">
                <Button className="flex-1" onClick={handleSave} disabled={processing === "save"}>
                  {processing === "save" ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <CheckCircle className="h-4 w-4 mr-2" />}
                  Enregistrer la facture
                </Button>
                <Button variant="outline" onClick={resetForm}>Réinitialiser</Button>
              </div>
            </div>
          </div>
        </TabsContent>

        {/* ── Fournisseurs tiers ── */}
        <TabsContent value="tiers" className="mt-4">
          <Card><CardContent className="p-0">
            <Table>
              <TableHeader><TableRow><TableHead>Nom</TableHead><TableHead>ICE</TableHead><TableHead>Email</TableHead><TableHead>Téléphone</TableHead></TableRow></TableHeader>
              <TableBody>
                {fournisseurs.length === 0
                  ? <TableRow><TableCell colSpan={4} className="text-center py-10 text-muted-foreground">
                      <Building2 className="h-8 w-8 mx-auto mb-2 opacity-30" />Aucun fournisseur
                    </TableCell></TableRow>
                  : fournisseurs.map(f => (
                    <TableRow key={f.id}>
                      <TableCell className="font-medium">{f.nom}</TableCell>
                      <TableCell className="font-mono text-xs">{f.ice ?? "—"}</TableCell>
                      <TableCell className="text-sm">{f.email ?? "—"}</TableCell>
                      <TableCell className="text-sm">{f.telephone ?? "—"}</TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </CardContent></Card>
        </TabsContent>
      </Tabs>

      {/* Modal fournisseur */}
      <Dialog open={openFourn} onOpenChange={setOpenFourn}>
        <DialogContent>
          <DialogHeader><DialogTitle>Nouveau fournisseur</DialogTitle></DialogHeader>
          <div className="space-y-3">
            <div className="space-y-2"><Label>Nom *</Label><Input value={formFourn.nom} onChange={e => setFormFourn({ ...formFourn, nom: e.target.value })} /></div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2"><Label>ICE</Label><Input value={formFourn.ice} onChange={e => setFormFourn({ ...formFourn, ice: e.target.value })} /></div>
              <div className="space-y-2"><Label>Email</Label><Input type="email" value={formFourn.email} onChange={e => setFormFourn({ ...formFourn, email: e.target.value })} /></div>
            </div>
            <div className="space-y-2"><Label>Téléphone</Label><Input value={formFourn.telephone} onChange={e => setFormFourn({ ...formFourn, telephone: e.target.value })} /></div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpenFourn(false)}>Annuler</Button>
            <Button onClick={async () => {
              if (!formFourn.nom) return;
              await supabase.from("fournisseurs").insert({ dossier_id: dossierId, ...formFourn, ice: formFourn.ice || null });
              setOpenFourn(false); setFormFourn({ nom: "", ice: "", email: "", telephone: "" }); load();
              toast.success("Fournisseur créé");
            }}>Créer</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
